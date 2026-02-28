/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/fused_router.h>

#include "../extensions.h"
#include "xla/ffi/api/c_api.h"

namespace transformer_engine {
namespace jax {

enum class ScoreFunction : int64_t {
  kSigmoid = 0,
  kSoftmax = 1,
};

// Compute num_tokens as the product of all dimensions except the last (num_experts).
// Supports arbitrary-rank inputs (e.g., [B, S, E] or [num_tokens, E]).
template <typename Dims>
static int compute_num_tokens(const Dims &dims) {
  int num_tokens = 1;
  for (size_t i = 0; i + 1 < dims.size(); ++i) {
    num_tokens *= static_cast<int>(dims[i]);
  }
  return num_tokens;
}

// ============================================================================
// Fused Top-K with Score Function - Forward
// ============================================================================

Error_Type FusedTopkWithScoreFunctionForwardFFI(
    cudaStream_t stream,
    Buffer_Type logits_buf,        // [num_tokens, num_experts]
    Buffer_Type expert_bias_buf,   // [num_experts] or empty
    Result_Type probs_buf,         // [num_tokens, num_experts] (or scores when compute_aux_scores)
    Result_Type routing_map_buf,   // [num_tokens, num_experts]
    Result_Type intermediate_buf,  // [num_tokens, num_experts]
    int64_t topk, int64_t use_pre_softmax, int64_t num_groups, int64_t group_topk,
    double scaling_factor, int64_t score_function, int64_t compute_aux_scores) {
  auto dtype = convert_ffi_datatype_to_te_dtype(logits_buf.element_type());
  auto dims = logits_buf.dimensions();
  auto num_tokens = compute_num_tokens(dims);
  auto num_experts = static_cast<int>(dims[dims.size() - 1]);

  auto *logits = logits_buf.untyped_data();
  auto *expert_bias = expert_bias_buf.untyped_data();
  auto *probs = probs_buf->untyped_data();
  auto *routing_map = routing_map_buf->untyped_data();
  auto *intermediate = intermediate_buf->untyped_data();

  auto flat_shape =
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)};
  auto logits_tensor = TensorWrapper(logits, flat_shape, dtype);
  auto probs_tensor = TensorWrapper(probs, flat_shape, dtype);
  auto routing_map_tensor = TensorWrapper(routing_map, flat_shape, DType::kByte);
  auto intermediate_tensor = TensorWrapper(intermediate, flat_shape, dtype);

  if (compute_aux_scores) {
    nvte_fused_score_for_moe_aux_loss_forward(
        logits_tensor.data(), num_tokens, num_experts, static_cast<int>(topk),
        static_cast<int>(score_function), probs_tensor.data(), routing_map_tensor.data(),
        intermediate_tensor.data(), stream);
  } else {
    auto bias_dims = expert_bias_buf.dimensions();
    auto expert_bias_tensor =
        (bias_dims.size() > 0 && bias_dims[0] > 0)
            ? TensorWrapper(expert_bias, std::vector<size_t>{static_cast<size_t>(bias_dims[0])},
                            convert_ffi_datatype_to_te_dtype(expert_bias_buf.element_type()))
            : TensorWrapper();

    nvte_fused_topk_with_score_function_forward(
        logits_tensor.data(), num_tokens, num_experts, static_cast<int>(topk),
        static_cast<int>(use_pre_softmax), static_cast<int>(num_groups),
        static_cast<int>(group_topk), static_cast<float>(scaling_factor),
        static_cast<int>(score_function), expert_bias_tensor.data(), probs_tensor.data(),
        routing_map_tensor.data(), intermediate_tensor.data(), stream);
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedTopkWithScoreFunctionForwardHandler,
                              FusedTopkWithScoreFunctionForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // logits
                                  .Arg<Buffer_Type>()      // expert_bias
                                  .Ret<Buffer_Type>()      // probs (or scores)
                                  .Ret<Buffer_Type>()      // routing_map
                                  .Ret<Buffer_Type>()      // intermediate_output
                                  .Attr<int64_t>("topk")
                                  .Attr<int64_t>("use_pre_softmax")
                                  .Attr<int64_t>("num_groups")
                                  .Attr<int64_t>("group_topk")
                                  .Attr<double>("scaling_factor")
                                  .Attr<int64_t>("score_function")
                                  .Attr<int64_t>("compute_aux_scores"),
                              FFI_CudaGraph_Traits);

// ============================================================================
// Fused Top-K with Score Function - Backward
// ============================================================================

Error_Type FusedTopkWithScoreFunctionBackwardFFI(
    cudaStream_t stream,
    Buffer_Type routing_map_buf,   // [num_tokens, num_experts] (unused when compute_aux_scores)
    Buffer_Type intermediate_buf,  // [num_tokens, num_experts]
    Buffer_Type grad_probs_buf,   // [num_tokens, num_experts] (grad_scores when compute_aux_scores)
    Result_Type grad_logits_buf,  // [num_tokens, num_experts]
    int64_t topk, int64_t use_pre_softmax, double scaling_factor, int64_t score_function,
    int64_t compute_aux_scores) {
  auto dtype = convert_ffi_datatype_to_te_dtype(intermediate_buf.element_type());
  auto dims = intermediate_buf.dimensions();
  auto num_tokens = compute_num_tokens(dims);
  auto num_experts = static_cast<int>(dims[dims.size() - 1]);

  auto flat_shape =
      std::vector<size_t>{static_cast<size_t>(num_tokens), static_cast<size_t>(num_experts)};

  auto intermediate_tensor = TensorWrapper(intermediate_buf.untyped_data(), flat_shape, dtype);
  auto grad_probs_tensor = TensorWrapper(grad_probs_buf.untyped_data(), flat_shape, dtype);
  auto grad_logits_tensor = TensorWrapper(grad_logits_buf->untyped_data(), flat_shape, dtype);

  if (compute_aux_scores) {
    nvte_fused_score_for_moe_aux_loss_backward(intermediate_tensor.data(), grad_probs_tensor.data(),
                                               num_tokens, num_experts, static_cast<int>(topk),
                                               static_cast<int>(score_function),
                                               grad_logits_tensor.data(), stream);
  } else {
    auto routing_map_tensor =
        TensorWrapper(routing_map_buf.untyped_data(), flat_shape, DType::kByte);

    nvte_fused_topk_with_score_function_backward(
        routing_map_tensor.data(), intermediate_tensor.data(), grad_probs_tensor.data(), num_tokens,
        num_experts, static_cast<int>(topk), static_cast<int>(use_pre_softmax),
        static_cast<float>(scaling_factor), static_cast<int>(score_function),
        grad_logits_tensor.data(), stream);
  }

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedTopkWithScoreFunctionBackwardHandler,
                              FusedTopkWithScoreFunctionBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // routing_map
                                  .Arg<Buffer_Type>()      // intermediate_output
                                  .Arg<Buffer_Type>()      // grad_probs
                                  .Ret<Buffer_Type>()      // grad_logits
                                  .Attr<int64_t>("topk")
                                  .Attr<int64_t>("use_pre_softmax")
                                  .Attr<double>("scaling_factor")
                                  .Attr<int64_t>("score_function")
                                  .Attr<int64_t>("compute_aux_scores"),
                              FFI_CudaGraph_Traits);

// ============================================================================
// Fused MoE Aux Loss - Forward
// ============================================================================

Error_Type FusedMoEAuxLossForwardFFI(cudaStream_t stream,
                                     Buffer_Type probs_buf,              // [num_rows, num_cols]
                                     Buffer_Type tokens_per_expert_buf,  // [num_experts]
                                     Result_Type aux_loss_buf,           // scalar
                                     Result_Type const_buf,              // scalar
                                     int64_t total_num_tokens, int64_t num_experts, int64_t topk,
                                     double coeff) {
  auto dtype = convert_ffi_datatype_to_te_dtype(probs_buf.element_type());
  auto probs_dims = probs_buf.dimensions();
  auto num_rows = static_cast<int>(probs_dims[0]);
  auto num_cols = static_cast<int>(probs_dims[1]);

  auto probs_shape =
      std::vector<size_t>{static_cast<size_t>(num_rows), static_cast<size_t>(num_cols)};
  auto tpe_dtype = convert_ffi_datatype_to_te_dtype(tokens_per_expert_buf.element_type());
  auto tpe_shape = std::vector<size_t>{static_cast<size_t>(num_experts)};
  auto scalar_shape = std::vector<size_t>{1};

  auto probs_tensor = TensorWrapper(probs_buf.untyped_data(), probs_shape, dtype);
  auto tpe_tensor = TensorWrapper(tokens_per_expert_buf.untyped_data(), tpe_shape, tpe_dtype);
  auto aux_loss_tensor = TensorWrapper(aux_loss_buf->untyped_data(), scalar_shape, dtype);
  auto const_buf_tensor = TensorWrapper(const_buf->untyped_data(), scalar_shape, DType::kFloat32);

  nvte_fused_moe_aux_loss_forward(
      probs_tensor.data(), tpe_tensor.data(), static_cast<int>(total_num_tokens),
      static_cast<int>(num_experts), num_rows, num_cols, static_cast<int>(topk),
      static_cast<float>(coeff), aux_loss_tensor.data(), const_buf_tensor.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedMoEAuxLossForwardHandler, FusedMoEAuxLossForwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // probs
                                  .Arg<Buffer_Type>()      // tokens_per_expert
                                  .Ret<Buffer_Type>()      // aux_loss
                                  .Ret<Buffer_Type>()      // Const_buf
                                  .Attr<int64_t>("total_num_tokens")
                                  .Attr<int64_t>("num_experts")
                                  .Attr<int64_t>("topk")
                                  .Attr<double>("coeff"),
                              FFI_CudaGraph_Traits);

// ============================================================================
// Fused MoE Aux Loss - Backward
// ============================================================================

Error_Type FusedMoEAuxLossBackwardFFI(cudaStream_t stream,
                                      Buffer_Type const_buf_in,           // scalar float32
                                      Buffer_Type tokens_per_expert_buf,  // [num_experts]
                                      Buffer_Type grad_aux_loss_buf,      // scalar
                                      Result_Type grad_probs_buf,         // [num_rows, num_cols]
                                      int64_t num_rows, int64_t num_cols) {
  auto grad_dtype = convert_ffi_datatype_to_te_dtype(grad_aux_loss_buf.element_type());
  auto tpe_dtype = convert_ffi_datatype_to_te_dtype(tokens_per_expert_buf.element_type());

  auto scalar_shape = std::vector<size_t>{1};
  auto tpe_dims = tokens_per_expert_buf.dimensions();
  auto tpe_shape = std::vector<size_t>{static_cast<size_t>(tpe_dims[0])};
  auto grad_probs_shape =
      std::vector<size_t>{static_cast<size_t>(num_rows), static_cast<size_t>(num_cols)};

  auto const_buf_tensor = TensorWrapper(const_buf_in.untyped_data(), scalar_shape, DType::kFloat32);
  auto tpe_tensor = TensorWrapper(tokens_per_expert_buf.untyped_data(), tpe_shape, tpe_dtype);
  auto grad_aux_loss_tensor =
      TensorWrapper(grad_aux_loss_buf.untyped_data(), scalar_shape, grad_dtype);
  auto grad_probs_tensor =
      TensorWrapper(grad_probs_buf->untyped_data(), grad_probs_shape, grad_dtype);

  nvte_fused_moe_aux_loss_backward(const_buf_tensor.data(), tpe_tensor.data(),
                                   static_cast<int>(num_rows), static_cast<int>(num_cols),
                                   grad_aux_loss_tensor.data(), grad_probs_tensor.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(FusedMoEAuxLossBackwardHandler, FusedMoEAuxLossBackwardFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // Const_buf
                                  .Arg<Buffer_Type>()      // tokens_per_expert
                                  .Arg<Buffer_Type>()      // grad_aux_loss
                                  .Ret<Buffer_Type>()      // grad_probs
                                  .Attr<int64_t>("num_rows")
                                  .Attr<int64_t>("num_cols"),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine
